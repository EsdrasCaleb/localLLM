package org.templateit;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.log4j.Logger;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFCellStyle;
import org.apache.poi.hssf.usermodel.HSSFDataFormatter;
import org.apache.poi.hssf.usermodel.HSSFFont;
import org.apache.poi.hssf.usermodel.HSSFFormulaEvaluator;
import org.apache.poi.hssf.usermodel.HSSFRichTextString;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.util.CellRangeAddress;
import com.lowagie.text.BadElementException;
import com.lowagie.text.Chunk;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.PageSize;
import com.lowagie.text.Phrase;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;
import com.lowagie.text.pdf.PdfPTable;

public class PdfWriter_writePdf_1_1_Test {

    @Test
    public void testWritePdf() throws Exception {
        // Create a dummy HSSFWorkbook and OutputStream
        HSSFWorkbook workbook = new HSSFWorkbook();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        // Create an instance of PdfWriter
        PdfWriter pdfWriter = new PdfWriter(workbook);
        // Call the writePdf method
        pdfWriter.writePdf(out);
        // Check if the output stream is not empty
        assertTrue(out.size() > 0);
        // Reset the output stream
        out.reset();
        // Create another instance of PdfWriter with the same workbook
        PdfWriter pdfWriter2 = new PdfWriter(workbook);
        // Call the writePdf method again
        pdfWriter2.writePdf(out);
        // Check if the output stream is not empty
        assertTrue(out.size() > 0);
        // Verify that the two instances are not the same
        assertNotSame(pdfWriter, pdfWriter2);
    }
}
