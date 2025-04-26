package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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

@ExtendWith(MockitoExtension.class)
public class NFeCalculatorImpl_calculate_0_2_Test {

    @Mock
    private ICMS icmsMock;

    @InjectMocks
    private NFeCalculatorImpl nfeCalculator;

    @Test
    public void calculateTest() {
        BigDecimal vBc = new BigDecimal("100");
        BigDecimal pICMS = new BigDecimal("10");
        BigDecimal vICMS = new BigDecimal("11");
        when(icmsMock.getVBc()).thenReturn(vBc);
        when(icmsMock.getPICMS()).thenReturn(pICMS);
        when(icmsMock.getVICMS()).thenReturn(vICMS);
        when(icmsMock.getModBC()).thenReturn(ModBC.MARGEM.getValue());
        BigDecimal result = nfeCalculator.calculate(icmsMock);
        assertEquals(vICMS, result);
    }
}
