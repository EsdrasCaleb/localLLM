package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.util.Map;
import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.util.Hashtable;

class RIFImplementationManager_registerImplementation_1_0_Test {

    @Test
    void registerImplementation_withValidInput_storesImplementation() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        Class<String> stringClass = String.class;
        String someString = "someValue";
        // Act
        manager.registerImplementation(stringClass, someString);
        // Correctly access the private field using reflection
        Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
        implementationMapField.setAccessible(true);
        Map<String, String> implementationMap = (Map<String, String>) implementationMapField.get(manager);
        String retrievedImplementation = implementationMap.get(stringClass.getName());
        assertEquals(someString, retrievedImplementation);
    }
}
