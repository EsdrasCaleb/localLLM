package org.templateit;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.apache.log4j.Logger;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFConditionalFormatting;
import org.apache.poi.hssf.usermodel.HSSFPrintSetup;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFSheetConditionalFormatting;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.util.CellRangeAddress;
import org.apache.poi.hssf.util.PaneInformation;
import org.templateit.util.FormulaUtil;

class TemplateProcessor_process_0_0_Test {

    @Test
    void testProcess(TemplateProcessor templateProcessor) throws IOException {
        // Given
        File outputWorkbook = new File("output.xlsx");
        TemplateProcessor templateProcessorMock = mock(TemplateProcessor.class);
        // When
        templateProcessor.process(new Iterator<String[]>() {

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public String[] next() {
                return new String[] { "Sheet1", "Sheet2" };
            }
        }, outputWorkbook);
        // Then
        // Verify the process method was called with the correct arguments and that it returned the correct output
        verify(templateProcessorMock).process(any(Iterator.class), any(File.class));
        assertEquals("output.xlsx", outputWorkbook.getAbsolutePath());
    }
}
