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

@ExtendWith(MockitoExtension.class)
public class FormulaUtil_offsetRelativeReferences_0_0_Test {

    // Test class
    @Test
    public void testOffsetRelativeReferences() {
        HSSFWorkbook wb = new HSSFWorkbook();
        String formula = "IF(A1,1,1+B2)";
        String newFormula = FormulaUtil.offsetRelativeReferences(wb, formula, 1, 2);
        assertEquals("IF(A1,1,2+B2)", newFormula);
    }
}
