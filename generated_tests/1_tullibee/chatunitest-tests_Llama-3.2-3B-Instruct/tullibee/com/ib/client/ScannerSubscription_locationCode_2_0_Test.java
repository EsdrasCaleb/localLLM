package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    @Test
    public void testLocationCode() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.instrument("Instrument");
        scanner.locationCode();
        scanner.locationCode();
        scanner.locationCode();
        assertEquals("Instrument", scanner.locationCode());
    }

    @Test
    public void testLocationCodeWithNull() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.locationCode();
        assertEquals(null, scanner.locationCode());
    }

    @Test
    public void testLocationCodeWithEmptyString() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.instrument("");
        scanner.locationCode();
        assertEquals("", scanner.locationCode());
    }

    @Test
    public void testLocationCodeWithNonString() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.instrument("Instrument");
        scanner.locationCode();
        assertEquals("Instrument", scanner.locationCode());
    }
}
