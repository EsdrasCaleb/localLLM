package org.templateit;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.Set;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
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
import org.apache.log4j.Logger;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFConditionalFormatting;
import org.apache.poi.hssf.usermodel.HSSFPrintSetup;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFSheetConditionalFormatting;
import org.apache.poi.hssf.util.CellRangeAddress;
import org.apache.poi.hssf.util.PaneInformation;
import org.templateit.util.FormulaUtil;

@ExtendWith(MockitoExtension.class)
public class TemplateProcessor_process_1_0_Test {

    @Mock
    private ByteArrayOutputStream bos;

    @Mock
    private Set<String> protectedSheetNames;

    @Mock
    private HSSFWorkbook workbook;

    @Mock
    private TemplateWorkbook tWorkbook;

    @InjectMocks
    private TemplateProcessor templateProcessor;

    @BeforeEach
    public void setUp() throws IOException {
        templateProcessor = new TemplateProcessor(mock(InputStream.class));
    }

    @Test
    public void testProcess() throws IOException {
        Iterator<String[]> di = mock(Iterator.class);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        when(bos.toByteArray()).thenReturn(new byte[0]);
        templateProcessor.process(di, out);
        verify(bos).toByteArray();
        verify(workbook).write(out);
    }
}
