package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Hashtable;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class RIFImplementationManager_invoke_2_0_Test {

    private RIFImplementationManager manager;

    private TestInterface testImpl;

    @BeforeEach
    void setUp() throws Exception {
        manager = RIFImplementationManager.getInstance();
        testImpl = Mockito.mock(TestInterface.class);
        // Use reflection to set the implementationMap and methodsMap
        Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
        implementationMapField.setAccessible(true);
        Hashtable<String, Object> implementationMap = (Hashtable<String, Object>) implementationMapField.get(manager);
        implementationMap.put("TestInterface", testImpl);
        Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
        methodsMapField.setAccessible(true);
        Hashtable<Object, Map<String, Method>> methodsMap = (Hashtable<Object, Map<String, Method>>) methodsMapField.get(manager);
        methodsMap.clear();
    }

    @Test
    void testInvokeSuccess() throws Exception {
        when(testImpl.testMethod("param")).thenReturn("result");
        Object result = manager.invoke("TestInterface", "testMethod", new Object[] { "param" });
        assertEquals("result", result);
        verify(testImpl).testMethod("param");
    }

    @Test
    void testInvokeInterfaceNotRegistered() {
        RemoteException exception = assertThrows(RemoteException.class, () -> {
            manager.invoke("NonRegisteredInterface", "testMethod", new Object[] { "param" });
        });
        assertEquals("Not registered: NonRegisteredInterface", exception.getMessage());
    }

    @Test
    void testInvokeMethodNotFound() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            manager.invoke("TestInterface", "nonExistentMethod", new Object[] { "param" });
        });
        assertEquals("Unknown method nonExistentMethod on TestInterface", exception.getMessage());
    }

    @Test
    void testInvokeExceptionDuringInvocation() throws Exception {
        when(testImpl.testMethod("param")).thenThrow(new RuntimeException("Test exception"));
        RemoteException exception = assertThrows(RemoteException.class, () -> {
            manager.invoke("TestInterface", "testMethod", new Object[] { "param" });
        });
        assertEquals("Error occured while invoking TestInterface.testMethod: Test exception", exception.getMessage());
    }

    interface TestInterface {

        String testMethod(String param);
    }
}
