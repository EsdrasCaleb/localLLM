package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_22_4_Test {

    @Test
    public void testInstrument() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        String instrument = "AAPL";
        scannerSubscription.instrument(instrument);
        Mockito.verify(scannerSubscription).instrument(instrument);
    }
}
