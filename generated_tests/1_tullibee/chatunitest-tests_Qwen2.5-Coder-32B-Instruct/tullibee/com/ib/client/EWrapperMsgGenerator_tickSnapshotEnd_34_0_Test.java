package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickSnapshotEnd_34_0_Test {

    @Test
    public void testTickSnapshotEnd() throws Exception {
        // Create an instance of EWrapperMsgGenerator
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Use reflection to invoke the private method
        String result = invokePrivateMethod(eWrapperMsgGenerator, "tickSnapshotEnd", 12345);
        // Define the expected result
        String expected = "id=12345 =============== end ===============";
        // Assert that the result matches the expected output
        assertEquals(expected, result);
    }

    @SuppressWarnings("unchecked")
    private <T> T invokePrivateMethod(Object object, String methodName, Object... params) throws Exception {
        Class<?>[] paramTypes = new Class[params.length];
        for (int i = 0; i < params.length; i++) {
            paramTypes[i] = params[i].getClass();
            // Handle cases where the parameter type is a primitive wrapper and convert to primitive type
            if (paramTypes[i] == Integer.class) {
                paramTypes[i] = int.class;
            }
        }
        Method method = object.getClass().getDeclaredMethod(methodName, paramTypes);
        method.setAccessible(true);
        return (T) method.invoke(object, params);
    }
}
