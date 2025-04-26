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
    public void testContractDetailsEnd() throws Exception {
        // Arrange
        int reqId = 123;
        String expectedOutput = "reqId = 123 =============== end ===============";
        // Act
        String result = invokeContractDetailsEnd(reqId);
        // Assert
        assertEquals(expectedOutput, result);
    }

    private String invokeContractDetailsEnd(int reqId) throws Exception {
        // Use reflection to access the private method
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("contractDetailsEnd", int.class);
        method.setAccessible(true);
        return (String) method.invoke(new EWrapperMsgGenerator(), reqId);
    }
}
