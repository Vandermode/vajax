import math
import re

from vajax.utils.safe_eval import parse_spice_number

from .exc import ConverterError


class InstanceVMixin:
    def process_instance_v(self, lws, line, eol, annot, in_sub):
        """
        Process V instance (voltage source).

        Ngspice format:
            vname n+ n- dc=value
            vname n+ n- value
            vname n+ n- pwl(t1 v1 t2 v2 ...)
            vname n+ n- pulse(v1 v2 td tr tf pw per)

        VACASK format:
            vname (n+ n-) vsource dc=value
            vname (n+ n-) vsource type="pwl" wave=[t1, v1, t2, v2, ...]
        """
        annot["name"]
        parts = annot["words"]

        terminals = self.process_terminals(parts[:2])

        # Rest of line after terminals - reconstruct from original line
        # to preserve PWL parentheses content
        rest = parts[2:] if len(parts) > 2 else []

        model = "vsource"
        params = []

        if len(rest) == 0:
            # No parameters - default to dc=0
            params.append(("dc", "0"))
        else:
            # Reconstruct the rest of the line
            rest_str = " ".join(rest)

            if re.match(r"pwl\b", rest_str, re.IGNORECASE):
                coordinates = rest_str[3:].strip()
                if coordinates.startswith("("):
                    if not coordinates.endswith(")"):
                        raise ConverterError(f"{annot['name']}: expected PWL(time value ...)")
                    coordinates = coordinates[1:-1]
                params.extend([("type", '"pwl"'), ("wave", self.parse_pwl(coordinates))])
            elif rest_str.lower().startswith("dc="):
                params.append(("dc", self.format_value(rest_str[3:])))
            elif rest_str.lower().startswith("dc "):
                params.append(("dc", self.format_value(rest_str[3:].strip())))
            elif len(rest) == 1 and "=" not in rest[0]:
                # Single value - treat as DC
                params.append(("dc", self.format_value(rest[0])))
            elif "=" in rest[0]:
                # Parameter assignment
                p, v = rest[0].split("=", 1)
                params.append((p.strip(), self.format_value(v.strip())))
            else:
                # Treat first as DC value
                params.append(("dc", self.format_value(rest[0])))

        txt = lws + annot["output_name"] + " (" + " ".join(terminals) + ") " + model

        if len(params) > 0:
            param_strs = []
            for pn, pv in params:
                param_strs.append(f"{pn}={pv}")
            txt += " " + " ".join(param_strs)

        return txt

    def parse_pwl(self, pwl_str):
        """Convert numeric time/value pairs to VAJAX's flat numeric waveform."""
        tokens = re.split(r"[\s,]+", pwl_str.strip())
        if len(tokens) < 2 or len(tokens) % 2:
            raise ConverterError("PWL requires complete time/value pairs")

        values = []
        for i, token in enumerate(tokens):
            # Accept both SPICE scales (10n, 500m) and explicit units (10ns, 500mV).
            unit = "s" if i % 2 == 0 else "v"
            value, ok = parse_spice_number(token.lower().removesuffix(unit))
            if not ok or not math.isfinite(value):
                raise ConverterError(f"Invalid numeric PWL coordinate: {token}")
            values.append(value)
        return str(values)
