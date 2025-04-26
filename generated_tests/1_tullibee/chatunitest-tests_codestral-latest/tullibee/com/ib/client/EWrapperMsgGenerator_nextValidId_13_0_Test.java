package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_nextValidId_13_0_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testNextValidId() {
        int orderId = 123;
        String expectedMessage = "Next Valid Order ID: " + orderId;
        String actualMessage = eWrapperMsgGenerator.nextValidId(orderId);
        assertEquals(expectedMessage, actualMessage);
    }
}
