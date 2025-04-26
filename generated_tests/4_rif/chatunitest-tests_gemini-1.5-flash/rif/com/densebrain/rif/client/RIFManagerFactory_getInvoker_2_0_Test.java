package com.densebrain.rif.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.rmi.RemoteException;
import java.util.Hashtable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class RIFManagerFactory_getInvoker_2_0_Test {

    @Mock
    private RIFManager mockRIFManager;

    @InjectMocks
    private RIFManagerFactory rifManagerFactory;

    @Test
    void testGetInvokerHappyPath() throws RemoteException {
        // Arrange
        String testUrl = "testUrl";
        Class<?> testInterface = String.class;
        // create a mock invoker
        RIFInvokerImpl mockInvoker = new RIFInvokerImpl();
        when(mockRIFManager.getInvoker(testInterface)).thenReturn(mockInvoker);
        rifManagerFactory.managerMap.put(testUrl, mockRIFManager);
        // Act
        RIFInvoker invoker = rifManagerFactory.getInvoker(testUrl, testInterface);
        // Assert
        assertNotNull(invoker);
        verify(mockRIFManager).getInvoker(testInterface);
    }

    @Test
    void testGetInvokerNullManager() throws RemoteException {
        // Arrange
        String testUrl = "testUrl";
        Class<?> testInterface = String.class;
        // Act & Assert
        assertThrows(NullPointerException.class, () -> rifManagerFactory.getInvoker(testUrl, testInterface));
    }

    @Test
    void testGetInvokerRemoteException() throws RemoteException {
        // Arrange
        String testUrl = "testUrl";
        Class<?> testInterface = String.class;
        when(mockRIFManager.getInvoker(testInterface)).thenThrow(new RemoteException("test exception"));
        rifManagerFactory.managerMap.put(testUrl, mockRIFManager);
        // Act & Assert
        assertThrows(RemoteException.class, () -> rifManagerFactory.getInvoker(testUrl, testInterface));
        verify(mockRIFManager).getInvoker(testInterface);
    }

    static class RIFInvokerImpl implements RIFInvoker {
    }

    interface RIFInvoker {
    }

    static class RIFManagerImpl implements RIFManager {

        @Override
        public RIFInvoker getInvoker(Class interfaceClazz) throws RemoteException {
            return null;
        }
    }

    interface RIFManager {

        RIFInvoker getInvoker(Class interfaceClazz) throws RemoteException;
    }

    static class RIFManagerFactory {

        Hashtable<String, RIFManager> managerMap = new Hashtable<>();

        public RIFInvoker getInvoker(String url, Class<?> interfaceClazz) throws RemoteException {
            RIFManager manager = managerMap.get(url);
            if (manager == null) {
                throw new NullPointerException("RIFManager not found for URL: " + url);
            }
            return manager.getInvoker(interfaceClazz);
        }
    }
}
