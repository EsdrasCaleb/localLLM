package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_managedAccounts_24_4_Test {

    @Test
    public void testManagedAccounts() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String accountsList = "Account1,Account2,Account3";
        // Act
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("managedAccounts", String.class);
        String result = (String) method.invoke(eWrapperMsgGenerator, accountsList);
        // Assert
        assertEquals("Connected : The list of managed accounts are : [Account1,Account2,Account3]", result);
    }
}
