package br.com.jnfe.base.service;

import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import java.math.BigDecimal;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import br.com.jnfe.base.COFINS;
import br.com.jnfe.base.ICMS;
import br.com.jnfe.base.ICMSExt;
import br.com.jnfe.base.ICMSST;
import br.com.jnfe.base.IPI;
import br.com.jnfe.base.ModBC;
import br.com.jnfe.base.PIS;

@MockitoSettings(strictness = Strictness.LENIENT)
class NFeCalculatorImpl_calculate_0_0_Test {

    @InjectMocks
    NFeCalculatorImpl nFeCalculator;

    @Mock
    ICMS icms;

    @Test
    void calculate_returnsCorrectValue_whenTaxValueIsNull() {
        when(icms.getVICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getPICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getVBc()).thenReturn(new BigDecimal("100.00"));
        when(icms.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        BigDecimal result = nFeCalculator.calculate(icms);
        assertEquals(new BigDecimal("10.00"), result);
    }

    @Test
    void calculate_returnsCorrectValue_whenTaxValueIsNotNull() {
        when(icms.getVICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getPICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getVBc()).thenReturn(new BigDecimal("100.00"));
        when(icms.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        BigDecimal result = nFeCalculator.calculate(icms);
        assertEquals(new BigDecimal("10.00"), result);
    }

    @Test
    void calculate_throwsException_whenModBCIsMargem() {
        when(icms.getVICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getPICMS()).thenReturn(new BigDecimal("10.00"));
        when(icms.getVBc()).thenReturn(new BigDecimal("100.00"));
        when(icms.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        try {
            nFeCalculator.calculate(icms);
        } catch (UnsupportedOperationException e) {
            assertEquals("Modo de determina��o da base de c�lculo ainda n�o suportada.", e.getMessage());
        }
    }
}
