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

public class EWrapperMsgGenerator_contractDetailsEnd_18_0_Test {

    @Test
    public void testContractDetailsEnd() {
        // Setup
        MockitoAnnotations.openMocks(EWrapperMsgGenerator.class);
        // Create an instance of the class under test
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Call the method to be tested
        String result = eWrapperMsgGenerator.contractDetailsEnd(123);
        // Verify the method's output
        assertEquals("123=", result);
    }
}
