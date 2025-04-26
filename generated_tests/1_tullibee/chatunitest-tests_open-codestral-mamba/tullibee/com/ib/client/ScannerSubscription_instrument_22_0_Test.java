package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.ib.client.ScannerSubscription;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_instrument_22_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrument() {
        scannerSubscription.instrument("AAPL");
        assertEquals("AAPL", scannerSubscription.instrument());
    }
}
