package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_22_0_Test {

    @Test
    void testInstrumentNull() {
        ScannerSubscription sub = new ScannerSubscription();
        sub.instrument(null);
        assertNull(getPrivateField(sub, "m_instrument"));
    }

    @Test
    void testInstrumentEmpty() {
        ScannerSubscription sub = new ScannerSubscription();
        sub.instrument("");
        assertEquals("", getPrivateField(sub, "m_instrument"));
    }

    @Test
    void testInstrumentValid() {
        ScannerSubscription sub = new ScannerSubscription();
        String instrument = "AAPL";
        sub.instrument(instrument);
        assertEquals(instrument, getPrivateField(sub, "m_instrument"));
    }

    @Test
    void testInstrumentWhitespace() {
        ScannerSubscription sub = new ScannerSubscription();
        sub.instrument("   ");
        assertEquals("   ", getPrivateField(sub, "m_instrument"));
    }

    private Object getPrivateField(ScannerSubscription obj, String fieldName) {
        try {
            Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            // Or throw an exception, depending on your error handling strategy
            return null;
        }
    }
}
