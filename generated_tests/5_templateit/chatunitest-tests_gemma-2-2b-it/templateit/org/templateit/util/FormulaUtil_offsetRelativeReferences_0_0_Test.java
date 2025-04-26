package org.templateit.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.HSSFFormulaParser;
import org.apache.poi.hssf.record.formula.AreaPtg;
import org.apache.poi.hssf.record.formula.Ptg;
import org.apache.poi.hssf.record.formula.RefPtg;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

public class FormulaUtil_offsetRelativeReferences_0_0_Test {

    @Test
    void testOffsetRelativeReferences() {
        FormulaUtil formulaUtil = new FormulaUtil();
        HSSFWorkbook workbook = new HSSFWorkbook();
        String formula = "A1";
        int roff = 0;
        int coff = 0;
        String expected = "A1";
        String actual = formulaUtil.offsetRelativeReferences(workbook, formula, roff, coff);
        assertEquals(expected, actual);
    }
}
