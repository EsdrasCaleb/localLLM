package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Hashtable;
import java.util.Map;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RIFImplementationManager_invoke_2_0_Test {

    @InjectMocks
    private RIFImplementationManager manager;

    @Mock
    private Object mockImpl;

    @Mock
    private Method mockMethod;

    @BeforeEach
    void setUp() {
        // No need to get the instance, @InjectMocks handles it
        // manager = RIFImplementationManager.getInstance();  // Removed
        try {
            Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
            implementationMapField.setAccessible(true);
            ((Map<String, Object>) implementationMapField.get(manager)).put("MyInterface", mockImpl);
            Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
            methodsMapField.setAccessible(true);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private fields: " + e.getMessage());
        }
    }

    @Test
    void invoke_success() throws Exception {
        String interfaceName = "MyInterface";
        String methodName = "myMethod";
        Object[] params = new Object[] { 1, "test" };
        Mockito.when(mockMethod.invoke(mockImpl, params)).thenReturn("success");
        // Correctly create and populate the method map
        Map<String, Method> methodMap = new Hashtable<>();
        methodMap.put(methodName, mockMethod);
        try {
            Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
            methodsMapField.setAccessible(true);
            // Correct key
            ((Map<String, Map<String, Method>>) methodsMapField.get(manager)).put(interfaceName, methodMap);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        Object result = manager.invoke(interfaceName, methodName, params);
        assertEquals("success", result);
    }

    @Test
    void invoke_interfaceNotFound() {
        String interfaceName = "NonExistentInterface";
        String methodName = "someMethod";
        Object[] params = new Object[] {};
        assertThrows(RemoteException.class, () -> manager.invoke(interfaceName, methodName, params));
    }

    @Test
    void invoke_methodNotFound() {
        String interfaceName = "MyInterface";
        String methodName = "nonExistentMethod";
        Object[] params = new Object[] {};
        assertThrows(IllegalArgumentException.class, () -> manager.invoke(interfaceName, methodName, params));
    }

    @Test
    void invoke_invocationError() throws Exception {
        String interfaceName = "MyInterface";
        String methodName = "myMethod";
        Object[] params = new Object[] { 1, "test" };
        Map<String, Method> methodMap = new Hashtable<>();
        methodMap.put(methodName, mockMethod);
        try {
            Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
            methodsMapField.setAccessible(true);
            // Correct key
            ((Map<String, Map<String, Method>>) methodsMapField.get(manager)).put(interfaceName, methodMap);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        Mockito.doThrow(new RuntimeException("Simulated error")).when(mockMethod).invoke(mockImpl, params);
        assertThrows(RemoteException.class, () -> manager.invoke(interfaceName, methodName, params));
    }
}
