package org.templateit.util;

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.record.formula.AreaPtg;
import org.apache.poi.hssf.record.formula.Ptg;
import org.apache.poi.hssf.record.formula.RefPtg;
import org.templateit.util.FormulaUtil;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.HSSFFormulaParser;

public class FormulaUtil_offsetRelativeReferences_0_2_Test {

    @Test
    public void offsetRelativeReferences_positiveOffset() {
        HSSFWorkbook wb = Mockito.mock(HSSFWorkbook.class);
        String formula = "A1+B2";
        int roff = 2;
        int coff = 1;
        String expectedFormula = "C3+D4";
        String actualFormula = FormulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        assertEquals(expectedFormula, actualFormula);
    }

    @Test
    public void offsetRelativeReferences_negativeOffset() {
        HSSFWorkbook wb = Mockito.mock(HSSFWorkbook.class);
        String formula = "C3+D4";
        int roff = -2;
        int coff = -1;
        String expectedFormula = "A1+B2";
        String actualFormula = FormulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        assertEquals(expectedFormula, actualFormula);
    }

    @Test
    public void offsetRelativeReferences_zeroOffset() {
        HSSFWorkbook wb = Mockito.mock(HSSFWorkbook.class);
        String formula = "A1+B2";
        int roff = 0;
        int coff = 0;
        String expectedFormula = "A1+B2";
        String actualFormula = FormulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        assertEquals(expectedFormula, actualFormula);
    }

    @Test
    public void offsetRelativeReferences_emptyFormula() {
        HSSFWorkbook wb = Mockito.mock(HSSFWorkbook.class);
        String formula = "";
        int roff = 2;
        int coff = 1;
        String expectedFormula = "";
        String actualFormula = FormulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        assertEquals(expectedFormula, actualFormula);
    }
}
