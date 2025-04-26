package org.templateit;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
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

class TemplateProcessor_process_1_4_Test {

    @Test
    void process_validInput_success() throws IOException {
        // Arrange
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        InputStream is = new ByteArrayInputStream("some input".getBytes());
        TemplateProcessor processor = new TemplateProcessor(is);
        Iterator<String[]> di = mock(Iterator.class);
        OutputStream out = mock(OutputStream.class);
        // Act
        processor.process(di, out);
        // Assert
        verify(di, times(1)).next();
        verify(out, times(1)).write("some input".getBytes());
    }

    @Test
    void process_invalidInput_throwsIOException() throws IOException {
        // Arrange
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        InputStream is = new ByteArrayInputStream("some input".getBytes());
        TemplateProcessor processor = new TemplateProcessor(is);
        Iterator<String[]> di = mock(Iterator.class);
        OutputStream out = mock(OutputStream.class);
        doThrow(IOException.class).when(out).write("some input".getBytes());
        // Act & Assert
        assertThrows(IOException.class, () -> processor.process(di, out));
    }
}
