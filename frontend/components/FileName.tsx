import { Fragment } from "react";
import { breakParts } from "@/lib/citations";

/** A file name that wraps after underscores, dots and dashes instead of mid-word. */
export default function FileName({ name }: { name: string }) {
  return (
    <>
      {breakParts(name).map((part, i) => (
        <Fragment key={i}>
          {part}
          <wbr />
        </Fragment>
      ))}
    </>
  );
}
