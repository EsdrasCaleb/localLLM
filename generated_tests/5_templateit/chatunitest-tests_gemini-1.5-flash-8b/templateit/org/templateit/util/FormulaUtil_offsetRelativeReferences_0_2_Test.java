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
    public void offsetRelativeReferences_zeroOffset() {
        HSSFWorkbook wb = Mockito.mock(HSSFWorkbook.class);
        String formula = "A1+B2";
        int roff = 0;
        int coff = 0;
        String expectedFormula = "A1+B2";
        String actualFormula = FormulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        assertEquals(expectedFormula, actualFormula);
    }
}
