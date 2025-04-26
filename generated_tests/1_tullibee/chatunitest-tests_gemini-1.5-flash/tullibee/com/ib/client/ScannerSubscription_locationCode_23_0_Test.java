package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_23_0_Test {

    @Test
    void testLocationCodeNull() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode(null);
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        assertNull(field.get(scannerSubscription));
    }

    @Test
    void testLocationCodeEmpty() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("");
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        assertEquals("", field.get(scannerSubscription));
    }

    @Test
    void testLocationCodeNonEmpty() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String locationCode = "TestLocation";
        scannerSubscription.locationCode(locationCode);
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        assertEquals(locationCode, field.get(scannerSubscription));
    }

    @Test
    void testLocationCodeWhitespace() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("   ");
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        assertEquals("   ", field.get(scannerSubscription));
    }
}
