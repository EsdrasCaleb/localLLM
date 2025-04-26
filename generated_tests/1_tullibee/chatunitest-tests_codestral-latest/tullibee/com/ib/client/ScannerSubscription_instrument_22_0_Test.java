package com.ib.client;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_instrument_22_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrument() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedInstrument = "AAPL";
        // Act
        scannerSubscription.instrument(expectedInstrument);
        // Assert
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(scannerSubscription);
        assertEquals(expectedInstrument, actualInstrument);
    }

    @Test
    public void testInstrumentNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedInstrument = null;
        // Act
        scannerSubscription.instrument(expectedInstrument);
        // Assert
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(scannerSubscription);
        assertNull(actualInstrument);
    }
}
