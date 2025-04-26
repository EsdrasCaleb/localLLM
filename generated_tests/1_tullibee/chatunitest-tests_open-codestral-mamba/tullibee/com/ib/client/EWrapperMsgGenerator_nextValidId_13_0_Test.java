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
public class EWrapperMsgGenerator_nextValidId_13_0_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @Test
    public void testNextValidId() {
        int orderId = 123;
        String expectedResult = "Next Valid Order ID: " + orderId;
        String actualResult = eWrapperMsgGenerator.nextValidId(orderId);
        assertEquals(expectedResult, actualResult);
    }
}
