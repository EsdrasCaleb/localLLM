package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_22_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrumentSetsValue() {
        // Arrange
        String expectedInstrument = "AAPL";
        // Act
        scannerSubscription.instrument(expectedInstrument);
        // Assert
        String actualInstrument = getPrivateField(scannerSubscription, "m_instrument");
        assertEquals(expectedInstrument, actualInstrument);
    }

    @Test
    public void testInstrumentSetsNullValue() {
        // Arrange
        String expectedInstrument = null;
        // Act
        scannerSubscription.instrument(expectedInstrument);
        // Assert
        String actualInstrument = getPrivateField(scannerSubscription, "m_instrument");
        assertEquals(expectedInstrument, actualInstrument);
    }

    private String getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
