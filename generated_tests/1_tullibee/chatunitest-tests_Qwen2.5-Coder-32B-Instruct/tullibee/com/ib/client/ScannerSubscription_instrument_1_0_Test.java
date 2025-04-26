package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrument_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        instrumentField.set(scannerSubscription, null);
        // Act
        String result = scannerSubscription.instrument();
        // Assert
        assertNull(result);
    }

    @Test
    public void testInstrument_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testInstrument = "TEST_INSTRUMENT";
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        instrumentField.set(scannerSubscription, testInstrument);
        // Act
        String result = scannerSubscription.instrument();
        // Assert
        assertEquals(testInstrument, result);
    }
}
