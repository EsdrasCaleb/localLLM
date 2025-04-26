package com.densebrain.rif.client;

// Test class
import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Hashtable;

@RunWith(MockitoJUnitRunner.class)
public class RIFManagerFactory_getImpl_3_4_Test {

    @Test
    public void testGetImpl() throws RemoteException {
        Class interfaceClazz = RIFManager.class;
        RIFManagerFactory manager = Mockito.mock(RIFManagerFactory.class);
        Mockito.when(manager.getImpl(Mockito.anyString(), Mockito.any(Class.class))).thenReturn("mockImpl");
        RIFManagerFactory.getInstance().getImpl("url", interfaceClazz);
        Method getImplMethod = RIFManagerFactory.class.getDeclaredMethods()[0];
        try {
            getImplMethod.invoke(RIFManagerFactory.getInstance(), "url", interfaceClazz);
        } catch (IllegalArgumentException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }
}
