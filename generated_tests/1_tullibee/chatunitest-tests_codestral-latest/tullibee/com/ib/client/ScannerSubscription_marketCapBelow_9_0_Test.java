package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        double expectedMarketCapBelow = 1000.0;
        setPrivateField(scannerSubscription, "m_marketCapBelow", expectedMarketCapBelow);
        // Act
        double actualMarketCapBelow = scannerSubscription.marketCapBelow();
        // Assert
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow);
    }

    private void setPrivateField(Object obj, String fieldName, Object value) throws NoSuchFieldException, IllegalAccessException {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }
}
