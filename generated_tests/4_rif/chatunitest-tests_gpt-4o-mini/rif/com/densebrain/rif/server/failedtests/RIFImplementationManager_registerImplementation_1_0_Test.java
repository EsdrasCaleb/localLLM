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

public class RIFImplementationManager_registerImplementation_1_0_Test {

    private RIFImplementationManager manager;

    @BeforeEach
    public void setUp() {
        manager = RIFImplementationManager.getInstance();
    }

    @Test
    public void testRegisterImplementation() {
        // Arrange
        Class<Runnable> interfaceClazz = Runnable.class;
        Runnable implementation = mock(Runnable.class);
        // Act
        manager.registerImplementation(interfaceClazz, implementation);
        // Assert
        Hashtable<String, Object> implementationMap = getImplementationMap(manager);
        assertNotNull(implementationMap);
        assertEquals(implementation, implementationMap.get(interfaceClazz.getName()));
    }

    @Test
    public void testRegisterMultipleImplementations() {
        // Arrange
        Class<Runnable> interfaceClazz1 = Runnable.class;
        Class<AutoCloseable> interfaceClazz2 = AutoCloseable.class;
        Runnable implementation1 = mock(Runnable.class);
        AutoCloseable implementation2 = mock(AutoCloseable.class);
        // Act
        manager.registerImplementation(interfaceClazz1, implementation1);
        manager.registerImplementation(interfaceClazz2, implementation2);
        // Assert
        Hashtable<String, Object> implementationMap = getImplementationMap(manager);
        assertEquals(implementation1, implementationMap.get(interfaceClazz1.getName()));
        assertEquals(implementation2, implementationMap.get(interfaceClazz2.getName()));
    }

    @Test
    public void testRegisterImplementationWithNullImplementation() {
        // Arrange
        Class<Runnable> interfaceClazz = Runnable.class;
        // Act
        manager.registerImplementation(interfaceClazz, null);
        // Assert
        Hashtable<String, Object> implementationMap = getImplementationMap(manager);
        assertNull(implementationMap.get(interfaceClazz.getName()));
    }

    private Hashtable<String, Object> getImplementationMap(RIFImplementationManager manager) {
        try {
            java.lang.reflect.Field field = RIFImplementationManager.class.getDeclaredField("implementationMap");
            field.setAccessible(true);
            return (Hashtable<String, Object>) field.get(manager);
        } catch (Exception e) {
            fail("Failed to access implementationMap field: " + e.getMessage());
            // Unreachable, but required for compilation
            return null;
        }
    }
}
