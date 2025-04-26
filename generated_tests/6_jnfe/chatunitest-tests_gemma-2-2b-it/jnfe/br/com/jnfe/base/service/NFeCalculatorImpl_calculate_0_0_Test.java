package br.com.jnfe.base.service;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.math.BigDecimal;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import br.com.jnfe.base.COFINS;
import br.com.jnfe.base.ICMS;
import br.com.jnfe.base.ICMSExt;
import br.com.jnfe.base.ICMSST;
import br.com.jnfe.base.IPI;
import br.com.jnfe.base.ModBC;
import br.com.jnfe.base.PIS;

public class NFeCalculatorImpl_calculate_0_0_Test {

    @Test
    void calculate_withValidICMS_shouldReturnCalculatedTax() {
        ICMS icms = mock(ICMS.class);
        when(icms.getVICMS()).thenReturn(BigDecimal.valueOf(100));
        when(icms.getPICMS()).thenReturn(BigDecimal.valueOf(50));
        when(icms.getVBc()).thenReturn(BigDecimal.valueOf(10));
        NFeCalculatorImpl calculator = new NFeCalculatorImpl();
        BigDecimal result = calculator.calculate(icms);
        assertEquals(BigDecimal.valueOf(50), result);
    }

    @Test
    void calculate_withNullVICMS_shouldThrowUnsupportedOperationException() {
        ICMS icms = mock(ICMS.class);
        when(icms.getVICMS()).thenReturn(null);
        NFeCalculatorImpl calculator = new NFeCalculatorImpl();
        assertThrows(UnsupportedOperationException.class, () -> calculator.calculate(icms));
    }
}
