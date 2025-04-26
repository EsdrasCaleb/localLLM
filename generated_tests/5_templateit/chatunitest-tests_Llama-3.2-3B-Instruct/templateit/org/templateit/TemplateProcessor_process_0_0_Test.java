package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedList;
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

@ExtendWith(MockitoExtension.class)
public class TemplateProcessor_process_0_0_Test {

    @Mock
    private Logger logger;

    @Mock
    private File templateWorkbook;

    @Mock
    private Iterator<String[]> di;

    @Mock
    private File outputWorkbook;

    @InjectMocks
    private TemplateProcessor templateProcessor;

    @Test
    void testProcess() throws IOException {
        // Arrange
        when(di.hasNext()).thenReturn(true);
        when(di.next()).thenReturn(new String[] { "data1", "data2" });
        when(templateWorkbook.exists()).thenReturn(true);
        when(templateWorkbook.isDirectory()).thenReturn(true);
        when(outputWorkbook.exists()).thenReturn(false);
        // Act
        templateProcessor.process(di, outputWorkbook);
        // Assert
        verify(di).hasNext();
        verify(di).next();
        verify(templateWorkbook).exists();
        verify(templateWorkbook).isDirectory();
        verify(outputWorkbook).exists();
        verifyNoMoreInteractions(logger, di, templateWorkbook, outputWorkbook);
    }

    @Test
    void testProcessIOException() throws IOException {
        // Arrange
        when(di.hasNext()).thenReturn(true);
        when(di.next()).thenThrow(new IOException());
        // Act and Assert
        assertThrows(IOException.class, () -> templateProcessor.process(di, new File("output.xlsx")));
    }

    @Test
    void testProcessClose() throws IOException {
        // Arrange
        when(di.hasNext()).thenReturn(true);
        when(di.next()).thenReturn(new String[] { "data1", "data2" });
        when(templateWorkbook.exists()).thenReturn(true);
        when(templateWorkbook.isDirectory()).thenReturn(true);
        when(outputWorkbook.exists()).thenReturn(false);
        // Act
        templateProcessor.process(di, outputWorkbook);
        // Assert
        verify(di).hasNext();
        verify(di).next();
        verify(templateWorkbook).exists();
        verify(templateWorkbook).isDirectory();
        verify(outputWorkbook).exists();
        verifyNoMoreInteractions(logger, di, templateWorkbook, outputWorkbook);
    }
}
