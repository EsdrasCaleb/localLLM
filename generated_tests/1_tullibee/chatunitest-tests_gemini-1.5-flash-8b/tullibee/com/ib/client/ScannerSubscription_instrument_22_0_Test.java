package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_instrument_22_0_Test {

    @Test
    void instrument_setsInstrument() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String instrumentName = "AAPL";
        subscription.instrument(instrumentName);
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(subscription);
        assertEquals(instrumentName, actualInstrument);
    }

    @Test
    void instrument_nullInput_setsInstrumentToNull() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String instrumentName = null;
        subscription.instrument(instrumentName);
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(subscription);
        assertNull(actualInstrument);
    }

    @Test
    void instrument_emptyInput_setsInstrumentToEmpty() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String instrumentName = "";
        subscription.instrument(instrumentName);
        Field instrumentField = ScannerSubscription.class.getDeclaredField("m_instrument");
        instrumentField.setAccessible(true);
        String actualInstrument = (String) instrumentField.get(subscription);
        assertEquals("", actualInstrument);
    }
}
