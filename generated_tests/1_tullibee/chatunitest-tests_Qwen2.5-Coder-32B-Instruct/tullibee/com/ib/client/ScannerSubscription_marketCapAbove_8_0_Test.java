package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMarketCapAbove_DefaultValue() {
        double expected = Double.MAX_VALUE;
        double actual = scannerSubscription.marketCapAbove();
        assertEquals(expected, actual);
    }

    @Test
    public void testMarketCapAbove_SetValue() throws NoSuchFieldException, IllegalAccessException {
        double testValue = 1000000.0;
        Field field = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
        field.setAccessible(true);
        field.set(scannerSubscription, testValue);
        double expected = testValue;
        double actual = scannerSubscription.marketCapAbove();
        assertEquals(expected, actual);
    }
}
