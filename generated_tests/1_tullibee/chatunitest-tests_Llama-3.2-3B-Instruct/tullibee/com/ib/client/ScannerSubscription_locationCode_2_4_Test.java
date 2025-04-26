package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_4_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testLocationCode_ReturnsInitialValue() {
        scannerSubscription = new ScannerSubscription();
        assertEquals("", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGet() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("New York");
        assertEquals("New York", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetNull() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode(null);
        assertEquals(null, scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetEmptyString() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("");
        assertEquals("", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetBlank() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode(" ");
        assertEquals(" ", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetSpecialCharacters() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("New York!");
        assertEquals("New York!", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetLargeString() {
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.locationCode("This is a very long string that should not be used as a location code");
        assertEquals("This is a very long string that should not be used as a location code", scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCode_SetAndGetNullThrowsNullPointerException() {
        scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.locationCode(null));
    }
}
