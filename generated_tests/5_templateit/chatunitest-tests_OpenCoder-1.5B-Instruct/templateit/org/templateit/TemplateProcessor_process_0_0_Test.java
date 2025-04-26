package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
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

@ExtendWith(MockitoExtension.class)
public class TemplateProcessor_process_0_0_Test {

    @Mock
    private Iterator<String[]> di;

    @Mock
    private File outputWorkbook;

    @InjectMocks
    private TemplateProcessor processor;

    @Test
    public void testProcess() throws IOException {
        String[] data = { "Row1", "Row2" };
        when(di.next()).thenReturn(data);
        when(outputWorkbook.exists()).thenReturn(false);
        when(outputWorkbook.createNewFile()).thenReturn(true);
        processor.process(di, outputWorkbook);
        verify(di, times(1)).next();
        verify(outputWorkbook, times(1)).exists();
        verify(outputWorkbook, times(1)).createNewFile();
    }
}
