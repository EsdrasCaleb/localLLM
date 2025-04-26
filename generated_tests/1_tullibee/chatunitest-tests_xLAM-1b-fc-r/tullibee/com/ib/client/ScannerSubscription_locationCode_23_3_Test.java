package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_locationCode_23_3_Test {

    @Test
    public void testLocationCode() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.locationCode("TestLocationCode");
        assertEquals("TestLocationCode", subscription.locationCode());
    }
}
