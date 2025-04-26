package com.densebrain.rif.server;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.Hashtable;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class RIFImplementationManager_registerImplementation_1_0_Test {

    @Mock
    private Class<?> mockInterfaceClazz;

    @Mock
    private Object mockImplementation;

    private RIFImplementationManager manager;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        manager = RIFImplementationManager.getInstance();
        // Clear the static implementationMap before each test
        Field field = RIFImplementationManager.class.getDeclaredField("implementationMap");
        field.setAccessible(true);
        field.set(manager, new Hashtable<String, Object>());
    }

    @Test
    public void testRegisterImplementation() throws Exception {
        // Arrange
        String expectedKey = mockInterfaceClazz.getName();
        when(mockInterfaceClazz.getName()).thenReturn("com.example.SomeInterface");
        // Act
        manager.registerImplementation(mockInterfaceClazz, mockImplementation);
        // Assert
        Hashtable<String, Object> implementationMap = getImplementationMap();
        assertTrue(implementationMap.containsKey(expectedKey));
        assertSame(mockImplementation, implementationMap.get(expectedKey));
    }

    private Hashtable<String, Object> getImplementationMap() throws NoSuchFieldException, IllegalAccessException {
        Field field = RIFImplementationManager.class.getDeclaredField("implementationMap");
        field.setAccessible(true);
        return (Hashtable<String, Object>) field.get(manager);
    }
}
