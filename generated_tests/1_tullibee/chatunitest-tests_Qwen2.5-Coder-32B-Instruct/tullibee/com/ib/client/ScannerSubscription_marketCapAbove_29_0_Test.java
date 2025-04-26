package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAbove() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testCapValue = 1000000.0;
        Field marketCapAboveField = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
        marketCapAboveField.setAccessible(true);
        // When
        scannerSubscription.marketCapAbove(testCapValue);
        // Then
        double actualValue = (double) marketCapAboveField.get(scannerSubscription);
        assertEquals(testCapValue, actualValue, "The marketCapAbove value should be set correctly.");
    }
}
