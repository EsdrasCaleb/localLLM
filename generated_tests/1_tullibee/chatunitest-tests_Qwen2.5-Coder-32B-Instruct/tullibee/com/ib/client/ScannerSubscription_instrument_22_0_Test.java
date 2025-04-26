package com.ib.client;

import java.lang.reflect.Field;
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
    public void testInstrument() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testInstrument = "AAPL";
        // When
        scannerSubscription.instrument(testInstrument);
        // Then
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(scannerSubscription);
        assertEquals(testInstrument, actualInstrument);
    }
}
