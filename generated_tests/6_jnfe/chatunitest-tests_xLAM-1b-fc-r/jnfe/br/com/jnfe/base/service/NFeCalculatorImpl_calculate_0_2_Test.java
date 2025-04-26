package br.com.jnfe.base.service;

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

public class NFeCalculatorImpl_calculate_0_2_Test {

    @Mock
    private ICMS mockedICMS;

    @InjectMocks
    private NFeCalculatorImpl nfeCalculatorImpl;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testCalculate_whenICMSIsNull_thenThrowsUnsupportedOperationException() {
        when(mockedICMS.getVICMS()).thenReturn(null);
        when(mockedICMS.getPICMS()).thenReturn(BigDecimal.ONE);
        when(mockedICMS.getVBc()).thenReturn(BigDecimal.TEN);
        when(mockedICMS.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        assertThrows(UnsupportedOperationException.class, () -> nfeCalculatorImpl.calculate(mockedICMS));
    }

    @Test
    public void testCalculate_whenICMSIsNotNull_thenReturnsTaxValue() {
        when(mockedICMS.getVICMS()).thenReturn(BigDecimal.ONE);
        when(mockedICMS.getPICMS()).thenReturn(BigDecimal.ONE);
        when(mockedICMS.getVBc()).thenReturn(BigDecimal.TEN);
        when(mockedICMS.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        BigDecimal taxValue = nfeCalculatorImpl.calculate(mockedICMS);
        assertEquals(BigDecimal.TEN, taxValue);
    }
}
