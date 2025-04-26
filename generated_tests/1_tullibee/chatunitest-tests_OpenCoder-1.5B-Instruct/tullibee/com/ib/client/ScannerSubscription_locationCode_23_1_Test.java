package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_23_1_Test {

    @Test
    public void testLocationCode() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        String locationCode = "NYC";
        scannerSubscription.locationCode(locationCode);
        assertEquals(locationCode, scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCodeNoChange() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        String locationCode = "NYC";
        scannerSubscription.locationCode(locationCode);
        assertEquals(locationCode, scannerSubscription.locationCode());
    }

    @Test
    public void testLocationCodeNull() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        scannerSubscription.locationCode(null);
        assertNull(scannerSubscription.locationCode());
    }
}
