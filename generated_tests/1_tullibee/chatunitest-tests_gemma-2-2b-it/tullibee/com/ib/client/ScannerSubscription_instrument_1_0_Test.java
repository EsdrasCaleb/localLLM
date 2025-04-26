package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ScannerSubscription_instrument_1_0_Test {

    ScannerSubscription scannerSubscription;

    @Test
    void instrument() {
        scannerSubscription = new ScannerSubscription();
        String expected = "AAPL";
        String actual = scannerSubscription.instrument();
        assertEquals(expected, actual);
    }
}
