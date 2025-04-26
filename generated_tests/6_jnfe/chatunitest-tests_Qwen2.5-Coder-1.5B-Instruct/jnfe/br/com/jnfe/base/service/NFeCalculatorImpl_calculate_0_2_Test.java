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

class NFeCalculatorImpl_calculate_0_2_Test {

    @Mock
    private ICMS icms;

    @InjectMocks
    private NFeCalculatorImpl calculator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void shouldReturnTaxValueWhenTaxValueIsNotNull() {
        // Setup
        when(icms.getVICMS()).thenReturn(new BigDecimal(100));
        when(icms.getPICMS()).thenReturn(new BigDecimal(200));
        when(icms.getVBc()).thenReturn(new BigDecimal(300));
        // Execute
        BigDecimal result = calculator.calculate(icms);
        // Verify
        assertNotNull(result);
        assertEquals(new BigDecimal(100), result);
    }

    @Test
    public void shouldThrowUnsupportedOperationExceptionWhenModBCIsNotSupported() {
        // Setup
        when(icms.getVICMS()).thenReturn(null);
        when(icms.getPICMS()).thenReturn(new BigDecimal(200));
        when(icms.getVBc()).thenReturn(new BigDecimal(300));
        // Execute
        try {
            calculator.calculate(icms);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // Verify
            assertTrue(e.getMessage().contains("Modo de determina��o da base de c�lculo ainda n�o suportada."));
        }
    }
}
