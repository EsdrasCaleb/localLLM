package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.model.HSSFFormulaParser;
import org.apache.poi.hssf.record.formula.AreaPtg;
import org.apache.poi.hssf.record.formula.Ptg;
import org.apache.poi.hssf.record.formula.RefPtg;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

@ExtendWith(MockitoExtension.class)
public class FormulaUtil_offsetRelativeReferences_0_1_Test {

    @Mock
    private HSSFWorkbook wb;

    @InjectMocks
    private FormulaUtil formulaUtil;

    @Test
    public void testOffsetRelativeReferences() {
        // Arrange
        String formula = "A1";
        int roff = 1;
        int coff = 2;
        // Act
        String newFormula = formulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        // Assert
        assertEquals("A1", newFormula);
    }
}
