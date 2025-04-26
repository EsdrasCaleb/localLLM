package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.apache.poi.hssf.model.HSSFFormulaParser;
import org.apache.poi.hssf.record.formula.AreaPtg;
import org.apache.poi.hssf.record.formula.Ptg;
import org.apache.poi.hssf.record.formula.RefPtg;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings
public class FormulaUtil_offsetRelativeReferences_0_4_Test {

    @Mock
    private HSSFWorkbook wb;

    @Mock
    private HSSFFormulaParser parser;

    @InjectMocks
    private FormulaUtil formulaUtil;

    @Test
    public void testOffsetRelativeReferences() {
        // Arrange
        String formula = "=SUM($A$1:$A$10)";
        int roff = 2;
        int coff = 3;
        when(parser.parse(formula, wb)).thenReturn(new AreaPtg[] {/* expected AreaPtg */
        });
        // Act
        String newFormula = formulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        // Assert
        assertEquals("expected formula string", newFormula);
    }

    @Test
    public void testOffsetRelativeReferences_rowOffsetZero() {
        // Arrange
        String formula = "=SUM($A$1:$A$10)";
        int roff = 0;
        int coff = 3;
        when(parser.parse(formula, wb)).thenReturn(new AreaPtg[] {/* expected AreaPtg */
        });
        // Act
        String newFormula = formulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        // Assert
        assertEquals("expected formula string", newFormula);
    }

    @Test
    public void testOffsetRelativeReferences_columnOffsetZero() {
        // Arrange
        String formula = "=SUM($A$1:$A$10)";
        int roff = 2;
        int coff = 0;
        when(parser.parse(formula, wb)).thenReturn(new AreaPtg[] {/* expected AreaPtg */
        });
        // Act
        String newFormula = formulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        // Assert
        assertEquals("expected formula string", newFormula);
    }

    @Test
    public void testOffsetRelativeReferences_zeroOffset() {
        // Arrange
        String formula = "=SUM($A$1:$A$10)";
        int roff = 0;
        int coff = 0;
        when(parser.parse(formula, wb)).thenReturn(new AreaPtg[] {/* expected AreaPtg */
        });
        // Act
        String newFormula = formulaUtil.offsetRelativeReferences(wb, formula, roff, coff);
        // Assert
        assertEquals("expected formula string", newFormula);
    }
}
