package com.densebrain.rif.server;

import java.util.Hashtable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Map;

class RIFImplementationManager_registerImplementation_1_0_Test {

    @InjectMocks
    private RIFImplementationManager rifImplementationManager;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testRegisterImplementation() {
        // Arrange
        Class interfaceClazz = Runnable.class;
        Object implementation = mock(Runnable.class);
        // Act
        rifImplementationManager.registerImplementation(interfaceClazz, implementation);
        // Assert
        Hashtable<String, Object> implementationMap = getPrivateField("implementationMap");
        assertNotNull(implementationMap);
        assertEquals(implementation, implementationMap.get(interfaceClazz.getName()));
    }

    @SuppressWarnings("unchecked")
    private <T> T getPrivateField(String fieldName) {
        try {
            java.lang.reflect.Field field = RIFImplementationManager.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (T) field.get(rifImplementationManager);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
