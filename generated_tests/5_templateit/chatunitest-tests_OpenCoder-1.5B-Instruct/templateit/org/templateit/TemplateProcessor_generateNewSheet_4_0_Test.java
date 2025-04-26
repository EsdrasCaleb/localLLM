package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Iterator;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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
public class TemplateProcessor_generateNewSheet_4_0_Test {

    @Mock
    private TemplateWorkbook tWorkbook;

    @InjectMocks
    private TemplateProcessor templateProcessor;

    @Test
    public void testGenerateNewSheet() {
        // Arrange
        String templateSheetName = "templateSheet";
        String newSheetName = "newSheet";
        Iterator<String[]> di = new MockIterator();
        // Act
        String[] result = templateProcessor.generateNewSheet(templateSheetName, newSheetName, di);
        // Assert
        assertNotNull(result);
        assertEquals(3, result.length);
        verify(tWorkbook).getTemplateSheet(templateSheetName);
    }

    private static class MockIterator implements Iterator<String[]> {

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public String[] next() {
            return new String[] { "Column1", "Column2", "Column3" };
        }
    }
}
