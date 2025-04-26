package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_contractDetailsEnd_18_3_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator instance;

    @Test
    public void testContractDetailsEnd() {
        // Arrange
        int reqId = 1;
        // Act
        String result = instance.contractDetailsEnd(reqId);
        // Assert
        assertEquals("Contract details end.", result);
    }
}
