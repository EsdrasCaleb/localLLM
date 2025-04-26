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
public class RIFImplementationManager_invoke_2_0_Test {

    @Mock
    private Object mockImplementation;

    @Mock
    private Method mockMethod;

    private RIFImplementationManager manager;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        manager = RIFImplementationManager.getInstance();
        clearPrivateField(manager, "implementationMap");
        clearPrivateField(manager, "methodsMap");
        when(mockMethod.invoke(mockImplementation, any())).thenReturn("mockResult");
        when(mockMethod.getName()).thenReturn("testMethod");
        setPrivateField(manager, "implementationMap", "TestInterface", mockImplementation);
        Map<String, Method> methodMap = new Hashtable<>();
        methodMap.put("testMethod", mockMethod);
        setPrivateField(manager, "methodsMap", mockImplementation, methodMap);
    }

    private void clearPrivateField(Object object, String fieldName) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        if (field.getType().equals(Map.class)) {
            ((Map<?, ?>) field.get(object)).clear();
        }
    }

    private void setPrivateField(Object object, String fieldName, Object key, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        Map<Object, Object> map = (Map<Object, Object>) field.get(object);
        map.put(key, value);
    }

    @Test
    public void testInvoke_Success() throws Exception {
        Object result = manager.invoke("TestInterface", "testMethod", new Object[] {});
        assertEquals("mockResult", result);
        verify(mockMethod).invoke(mockImplementation, new Object[] {});
    }

    @Test
    public void testInvoke_InterfaceNotRegistered() {
        Exception exception = assertThrows(RemoteException.class, () -> {
            manager.invoke("NonExistentInterface", "testMethod", new Object[] {});
        });
        assertEquals("Not registered: NonExistentInterface", exception.getMessage());
    }

    @Test
    public void testInvoke_MethodNotFound() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            manager.invoke("TestInterface", "nonExistentMethod", new Object[] {});
        });
        assertEquals("Unknown method nonExistentMethod on TestInterface", exception.getMessage());
    }

    @Test
    public void testInvoke_MethodInvocationError() throws Exception {
        when(mockMethod.invoke(mockImplementation, any())).thenThrow(new IllegalAccessException("Test Exception"));
        Exception exception = assertThrows(RemoteException.class, () -> {
            manager.invoke("TestInterface", "testMethod", new Object[] {});
        });
        assertTrue(exception.getMessage().contains("Error occured while invoking TestInterface.testMethod: Test Exception"));
    }
}
