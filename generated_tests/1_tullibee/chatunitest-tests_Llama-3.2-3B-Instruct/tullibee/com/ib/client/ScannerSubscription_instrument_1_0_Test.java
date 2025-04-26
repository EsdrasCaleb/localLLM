package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_0_Test {

    @Test
    public void testInstrument() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.instrument("Test Instrument Code");
        assertEquals("Test Instrument Code", scannerSubscription.instrument());
    }

    @Test
    public void testInstrumentEmpty() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.instrument(null);
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.instrument());
    }

    @Test
    public void testInstrumentDefault() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.instrument();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.instrument());
    }
}
