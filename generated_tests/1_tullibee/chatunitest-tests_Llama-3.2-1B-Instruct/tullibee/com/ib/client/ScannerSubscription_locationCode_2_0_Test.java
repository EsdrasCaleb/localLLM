package com.ib.client;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@RunWith(MockitoJUnitRunner.class)
public class ScannerSubscription_locationCode_2_0_Test {

    @Mock
    private ScannerSubscription focal;

    @Test
    public void testLocationCode() {
        // Arrange
        String locationCode = "TEST";
        when(focal.locationCode()).thenReturn(locationCode);
        // Act
        String result = focal.locationCode();
        // Assert
        assertEquals(locationCode, result);
    }
}
